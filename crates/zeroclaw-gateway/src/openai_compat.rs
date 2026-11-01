//! OpenAI-compatible chat endpoint: `POST /v1/chat/completions` (and `GET /v1/models`).
//!
//! Lets any OpenAI client talk to a ZeroClaw agent. A request runs one gateway chat turn, the
//! same turn `/webhook` runs: its last user message is the turn's input. A turn has no transcript
//! of its own (the session only scopes memory recall), so the earlier user and assistant messages
//! in `messages` come with it as context, the most recent ones and each bounded.
//!
//! - Authentication is the webhook's: the gateway's pairing token, or `gateway.webhook_secret`.
//!   OpenAI clients send their key as `Authorization: Bearer <key>`; that key is checked as the
//!   pairing token and, when no `X-Webhook-Secret` header is present, as the webhook secret. At
//!   least one control must be configured, or every request is refused.
//! - `model` names the agent alias (`[agents.<alias>]`); an unknown or empty model uses the
//!   gateway's default agent.
//! - The session comes from the `X-Session-Id` header, else from the request's `user` field.
//! - `stream: true` answers with Server-Sent Events in the OpenAI chunk format; the reply arrives
//!   as one chunk once the turn ends (no token-by-token streaming yet).

use std::net::SocketAddr;

use axum::{
    Json,
    extract::{ConnectInfo, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{AppState, authorize_webhook_request, run_gateway_chat_with_tools, webhook_session_id};

/// The subset of an OpenAI chat completion request this endpoint reads.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// A string, or a list of content parts of which the text parts are used.
    #[serde(default)]
    pub content: Value,
}

/// The text of a message's content: a plain string, or the `text` parts of a list.
fn content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// The last user message.
fn last_user_message(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| content_text(&message.content))
        .filter(|text| !text.trim().is_empty())
}

/// How many earlier messages of the conversation are given to the turn, and how much of each.
const MAX_EARLIER_MESSAGES: usize = 12;
const MAX_EARLIER_CHARS: usize = 1000;

/// The turn's input: the last user message, preceded by the conversation before it. A gateway
/// turn starts from the agent's own prompt with no transcript, so an OpenAI client, which sends
/// the whole conversation each time, gets it read as context: the user's and the assistant's
/// earlier messages (system messages are the agent's own prompt's business), the most recent
/// ones, each bounded.
fn turn_input(messages: &[ChatMessage]) -> Option<String> {
    let last = messages
        .iter()
        .rposition(|message| message.role == "user")?;
    let current = last_user_message(messages)?;
    let earlier: Vec<String> = messages[..last]
        .iter()
        .filter_map(|message| {
            let speaker = match message.role.as_str() {
                "user" => "User",
                "assistant" => "Assistant",
                _ => return None,
            };
            let text = content_text(&message.content);
            let text = text.trim();
            (!text.is_empty()).then(|| {
                format!(
                    "{speaker}: {}",
                    text.chars().take(MAX_EARLIER_CHARS).collect::<String>()
                )
            })
        })
        .collect();
    if earlier.is_empty() {
        return Some(current);
    }
    let recent = &earlier[earlier.len().saturating_sub(MAX_EARLIER_MESSAGES)..];
    Some(format!(
        "[Earlier in this conversation, for context; answer the current message]\n{}\n\n[Current message]\n{current}",
        recent.join("\n")
    ))
}

/// A session id the gateway accepts (the same rules as `X-Session-Id`), derived from `user`.
fn session_from_user(user: &str) -> Option<String> {
    let id: String = user
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .take(128)
        .collect();
    (!id.is_empty()).then_some(id)
}

fn openai_error(status: StatusCode, message: &str, kind: &str) -> Response {
    let body = json!({ "error": { "message": message, "type": kind } });
    (status, Json(body)).into_response()
}

/// Copies an OpenAI-style bearer key into `X-Webhook-Secret` when the caller sent none, so the
/// webhook's own check accepts either the pairing token or the webhook secret as the API key.
fn with_bearer_as_webhook_secret(headers: &HeaderMap) -> HeaderMap {
    let mut headers = headers.clone();
    if headers.contains_key("X-Webhook-Secret") {
        return headers;
    }
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .and_then(|key| HeaderValue::from_str(key).ok());
    if let Some(key) = bearer {
        headers.insert("X-Webhook-Secret", key);
    }
    headers
}

pub async fn handle_chat_completions(
    State(state): State<AppState>,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let auth_headers = with_bearer_as_webhook_secret(&headers);
    match authorize_webhook_request(&state, peer_addr, &auth_headers) {
        Ok(verdict) if verdict.may_dispatch_sop() => {}
        Ok(_) => {
            return openai_error(
                StatusCode::UNAUTHORIZED,
                "Configure gateway.webhook_secret or pairing, and send it as the API key.",
                "authentication_error",
            );
        }
        Err((status, Json(body))) => return (status, Json(body)).into_response(),
    }

    let Json(request) = match body {
        Ok(body) => body,
        Err(error) => {
            return openai_error(StatusCode::BAD_REQUEST, &error.to_string(), "invalid_request_error");
        }
    };
    let Some(message) = turn_input(&request.messages) else {
        return openai_error(StatusCode::BAD_REQUEST, "No user message to answer.", "invalid_request_error");
    };

    let model = request.model.trim();
    let agent = {
        let config = state.config.read();
        (!model.is_empty() && config.agent(model).is_some()).then(|| model.to_string())
    };
    let session = webhook_session_id(&headers).or_else(|| request.user.as_deref().and_then(session_from_user));

    let reply = match run_gateway_chat_with_tools(&state, &message, session.as_deref(), agent.as_deref()).await {
        Ok(outcome) => outcome.response,
        Err(error) => {
            let sanitized = zeroclaw_providers::sanitize_api_error(&error.to_string());
            return openai_error(StatusCode::BAD_GATEWAY, &sanitized, "api_error");
        }
    };

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();
    let model_name = if model.is_empty() { "zeroclaw" } else { model };

    if request.stream {
        let chunk = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model_name,
            "choices": [{ "index": 0, "delta": { "role": "assistant", "content": reply }, "finish_reason": null }],
        });
        let end = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model_name,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
        });
        let body = format!("data: {chunk}\n\ndata: {end}\n\ndata: [DONE]\n\n");
        return ([(header::CONTENT_TYPE, "text/event-stream"), (header::CACHE_CONTROL, "no-cache")], body).into_response();
    }

    Json(json!({
        "id": id, "object": "chat.completion", "created": created, "model": model_name,
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": reply }, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
    }))
    .into_response()
}

/// `GET /v1/models`: the configured agents, as models.
pub async fn handle_models(State(state): State<AppState>) -> Response {
    let config = state.config.read();
    let data: Vec<Value> = config
        .agents
        .keys()
        .map(|alias| json!({ "id": alias, "object": "model", "owned_by": "zeroclaw" }))
        .collect();
    Json(json!({ "object": "list", "data": data })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: Value) -> ChatMessage {
        ChatMessage { role: role.to_string(), content }
    }

    #[test]
    fn a_first_message_is_the_input_as_is() {
        let messages = vec![
            message("system", json!("be brief")),
            message("user", json!("hello")),
        ];
        assert_eq!(turn_input(&messages).as_deref(), Some("hello"));
    }

    #[test]
    fn earlier_turns_come_as_context_before_the_current_message() {
        let messages = vec![
            message("system", json!("be brief")),
            message("user", json!("Delete Tears of Steel")),
            message("assistant", json!("Files too?")),
            message("user", json!("Everything")),
        ];
        let input = turn_input(&messages).expect("an input");
        assert!(
            input.contains("User: Delete Tears of Steel\nAssistant: Files too?"),
            "{input}"
        );
        assert!(input.ends_with("[Current message]\nEverything"), "{input}");
        assert!(!input.contains("be brief"), "{input}");
    }

    #[test]
    fn only_the_most_recent_earlier_messages_are_kept_each_bounded() {
        let mut messages: Vec<ChatMessage> = (0..20)
            .map(|i| message("user", json!(format!("m{i}"))))
            .collect();
        messages.push(message(
            "assistant",
            json!("q".repeat(MAX_EARLIER_CHARS + 10)),
        ));
        messages.push(message("user", json!("now")));
        let input = turn_input(&messages).expect("an input");
        assert!(!input.contains("User: m8\n"), "{input}");
        assert!(input.contains("User: m9\n"), "{input}");
        assert_eq!(input.matches('q').count(), MAX_EARLIER_CHARS);
    }

    #[test]
    fn takes_the_last_user_message() {
        let messages = vec![
            message("system", json!("be brief")),
            message("user", json!("first")),
            message("assistant", json!("ok")),
            message("user", json!([{ "type": "text", "text": "second" }, { "type": "image_url" }])),
        ];
        assert_eq!(last_user_message(&messages).as_deref(), Some("second"));
    }

    #[test]
    fn no_user_message_is_none() {
        assert!(last_user_message(&[message("system", json!("x"))]).is_none());
        assert!(last_user_message(&[message("user", json!("   "))]).is_none());
    }

    #[test]
    fn user_becomes_a_safe_session_id() {
        assert_eq!(session_from_user("phone:living room").as_deref(), Some("phonelivingroom"));
        assert!(session_from_user("@@@").is_none());
    }

    #[test]
    fn bearer_key_is_offered_as_webhook_secret() {
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer sk-test"));
        let out = with_bearer_as_webhook_secret(&headers);
        assert_eq!(out.get("X-Webhook-Secret").unwrap(), "sk-test");

        headers.insert("X-Webhook-Secret", HeaderValue::from_static("explicit"));
        let out = with_bearer_as_webhook_secret(&headers);
        assert_eq!(out.get("X-Webhook-Secret").unwrap(), "explicit");
    }
}
