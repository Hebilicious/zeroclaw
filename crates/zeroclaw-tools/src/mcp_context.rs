//! Helpers that format MCP resource/prompt content for safe injection into the
//! model context. All server-origin content is wrapped with an
//! `trust="untrusted-external"` provenance marker and run through the existing
//! secret-scrubbing/length-bounding used elsewhere for server-controlled text.

use crate::mcp_client::McpRegistry;
use crate::mcp_prompt::McpGetPromptResult;
use crate::mcp_resource::McpResourceContents;
use crate::tool_search::ToolAccessPolicy;
use std::sync::Arc;
use zeroclaw_config::schema::McpServerConfig;

/// One pinned MCP resource, rendered for the system prompt and attributed to
/// the exact-match selector name it was admitted under.
///
/// `key` is the `<server>__<uri>` name the access policy checked at assembly.
/// It is what a later narrowing of the caller's tool selector is matched
/// against, so admitted content can be withdrawn from the prompt when the
/// grant that admitted it is withdrawn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinnedResourceBlock {
    pub key: String,
    pub rendered: String,
}

/// Longest server `instructions` text kept in the prompt, in characters.
const MAX_SERVER_INSTRUCTIONS_CHARS: usize = 4000;

impl PinnedResourceBlock {
    /// Whether a caller allowed `allowed` (exact tool names) may still see this
    /// block. A resource block is keyed by its `<server>__<uri>` name and needs
    /// that name; a server-instructions block is keyed by the server's tool
    /// prefix `<server>__` and stays while any of that server's tools is allowed.
    #[must_use]
    pub fn is_admitted_by(&self, allowed: &[String]) -> bool {
        if self.key.ends_with("__") {
            allowed.iter().any(|name| name.starts_with(&self.key))
        } else {
            allowed.contains(&self.key)
        }
    }
}

/// The `instructions` a server sent at `initialize`, as a block admitted while
/// the caller may use at least one of that server's tools.
async fn server_instructions_block(
    registry: &Arc<McpRegistry>,
    server: &str,
    policy: Option<&ToolAccessPolicy>,
) -> Option<PinnedResourceBlock> {
    let instructions = registry.server_instructions(server).await?;
    let prefix = format!("{server}__");
    let usable = registry
        .tool_names()
        .iter()
        .any(|name| name.starts_with(&prefix) && policy.is_none_or(|p| p.is_tool_allowed(name)));
    if !usable {
        return None;
    }
    Some(PinnedResourceBlock {
        rendered: wrap_server_instructions(server, &instructions),
        key: prefix,
    })
}

/// Wrap a server's `instructions` in a provenance block, secret-scrubbed and
/// length-bounded like other server-origin text.
pub fn wrap_server_instructions(server: &str, instructions: &str) -> String {
    let scrubbed = zeroclaw_providers::scrub_secret_patterns(instructions);
    let bounded: String = scrubbed
        .chars()
        .take(MAX_SERVER_INSTRUCTIONS_CHARS)
        .collect();
    format!(
        "<mcp-server-instructions server=\"{}\" trust=\"untrusted-external\">\n{}\n</mcp-server-instructions>",
        attr_escape(server),
        bounded.trim_end()
    )
}

/// Read every policy-admitted pinned resource and return it as an attributed
/// block. [`render_pinned_resources_section`] turns the blocks into the prompt
/// section; keeping the blocks lets the holder prune them later.
pub async fn build_pinned_resource_blocks(
    registry: &Arc<McpRegistry>,
    configs: &[McpServerConfig],
    policy: Option<&ToolAccessPolicy>,
) -> Vec<PinnedResourceBlock> {
    let mut blocks = Vec::new();
    for cfg in configs {
        if let Some(block) = server_instructions_block(registry, &cfg.name, policy).await {
            blocks.push(block);
        }
        for uri in &cfg.pinned_resources {
            let prefixed = format!("{}__{}", cfg.name, uri);
            if let Some(p) = policy
                && !p.is_tool_allowed(&prefixed)
            {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({"pinned_uri": &prefixed})),
                    "mcp_context: pinned resource denied by access policy"
                );
                continue;
            }
            match registry.read_resource(&prefixed).await {
                Ok(contents) => {
                    blocks.push(PinnedResourceBlock {
                        rendered: wrap_resource_contents(&cfg.name, &prefixed, &contents),
                        key: prefixed,
                    });
                }
                Err(e) => {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({"pinned_uri": &prefixed})),
                        &format!("mcp_context: skipping pinned resource: {e}")
                    );
                }
            }
        }
    }
    blocks
}

/// Render attributed blocks (each server's instructions, then its pinned
/// resources) into the `## Pinned MCP Resources` prompt section. Empty input
/// renders to an empty string, so a holder whose last block was pruned emits no
/// section at all.
#[must_use]
pub fn render_pinned_resources_section(blocks: &[PinnedResourceBlock]) -> String {
    if blocks.is_empty() {
        return String::new();
    }
    let mut body = String::new();
    for block in blocks {
        body.push_str(&block.rendered);
        body.push('\n');
    }
    format!("## Pinned MCP Resources\n\n{body}")
}

/// Convenience for callers that only ever need the rendered section.
pub async fn build_pinned_resources_section(
    registry: &Arc<McpRegistry>,
    configs: &[McpServerConfig],
    policy: Option<&ToolAccessPolicy>,
) -> String {
    render_pinned_resources_section(&build_pinned_resource_blocks(registry, configs, policy).await)
}

/// Escape the few characters that would break our attribute quoting.
fn attr_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
}

/// Wrap `resources/read` contents in a provenance block. Text content is
/// scrubbed + length-bounded via `sanitize_api_error`; blobs are summarized,
/// never dumped.
pub fn wrap_resource_contents(
    server: &str,
    prefixed_uri: &str,
    contents: &McpResourceContents,
) -> String {
    let mut body = String::new();
    for c in &contents.contents {
        if let Some(text) = &c.text {
            body.push_str(&zeroclaw_providers::sanitize_api_error(text));
            body.push('\n');
        } else if let Some(blob) = &c.blob {
            let mime = c.mime_type.as_deref().unwrap_or("application/octet-stream");
            body.push_str(&format!(
                "[binary blob, {} bytes, mime={mime}]\n",
                blob.len()
            ));
        }
    }
    let mime = contents
        .contents
        .first()
        .and_then(|c| c.mime_type.clone())
        .unwrap_or_default();
    format!(
        "<mcp-resource server=\"{}\" uri=\"{}\" mime=\"{}\" trust=\"untrusted-external\">\n{}</mcp-resource>",
        attr_escape(server),
        attr_escape(prefixed_uri),
        attr_escape(&mime),
        body
    )
}

/// Render `prompts/get` messages into a labeled, untrusted-provenance block.
pub fn render_prompt_messages(
    server: &str,
    prefixed_name: &str,
    result: &McpGetPromptResult,
) -> String {
    let mut body = String::new();
    for m in &result.messages {
        let text = m
            .content
            .get("text")
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| m.content.to_string());
        let scrubbed = zeroclaw_providers::sanitize_api_error(&text);
        body.push_str(&format!("[{}] {}\n", attr_escape(&m.role), scrubbed));
    }
    format!(
        "<mcp-prompt server=\"{}\" name=\"{}\" trust=\"untrusted-external\">\n{}</mcp-prompt>",
        attr_escape(server),
        attr_escape(prefixed_name),
        body
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_instructions_block_stays_while_any_tool_of_its_server_is_allowed() {
        let instructions = PinnedResourceBlock {
            key: "media__".into(),
            rendered: wrap_server_instructions("media", "Find, then get."),
        };
        let resource = PinnedResourceBlock {
            key: "media__doc://a".into(),
            rendered: "r".into(),
        };
        let allowed = vec!["media__series_get".to_string()];
        assert!(instructions.is_admitted_by(&allowed));
        assert!(!resource.is_admitted_by(&allowed));
        assert!(!instructions.is_admitted_by(&["remote__devices_list".to_string()]));
        // A server named like another's prefix does not borrow its grant.
        assert!(!instructions.is_admitted_by(&["mediaplus__x".to_string()]));
        assert!(!instructions.is_admitted_by(&[]));
    }

    #[test]
    fn server_instructions_are_marked_untrusted_escaped_and_bounded() {
        let wrapped =
            wrap_server_instructions("a\"b", &"z".repeat(MAX_SERVER_INSTRUCTIONS_CHARS + 50));
        assert!(wrapped.starts_with(
            "<mcp-server-instructions server=\"a&quot;b\" trust=\"untrusted-external\">"
        ));
        assert!(wrapped.ends_with("</mcp-server-instructions>"));
        assert_eq!(wrapped.matches('z').count(), MAX_SERVER_INSTRUCTIONS_CHARS);
    }

    #[test]
    fn rendered_section_is_the_attributed_blocks_in_order_and_empty_when_pruned() {
        let block = |key: &str| PinnedResourceBlock {
            key: key.into(),
            rendered: format!("<mcp-resource uri=\"{key}\">body</mcp-resource>"),
        };
        let mut blocks = vec![block("srv__a"), block("srv__b")];
        let section = render_pinned_resources_section(&blocks);
        assert_eq!(
            section,
            "## Pinned MCP Resources\n\n\
             <mcp-resource uri=\"srv__a\">body</mcp-resource>\n\
             <mcp-resource uri=\"srv__b\">body</mcp-resource>\n"
        );
        // Pruning by key is what a holder does when the caller's selector
        // narrows; the last block gone means no section at all.
        blocks.retain(|b| b.key == "srv__b");
        assert!(render_pinned_resources_section(&blocks).contains("srv__b"));
        assert!(!render_pinned_resources_section(&blocks).contains("srv__a"));
        blocks.clear();
        assert_eq!(render_pinned_resources_section(&blocks), "");
    }
    use crate::mcp_client::McpRegistry;
    use crate::mcp_prompt::{McpGetPromptResult, McpPromptMessage};
    use crate::mcp_resource::{McpResourceContent, McpResourceContents};
    use tokio;
    use zeroclaw_config::schema::McpServerConfig;

    #[tokio::test]
    async fn pinned_section_empty_for_empty_registry() {
        let registry = std::sync::Arc::new(McpRegistry::connect_all(&[]).await.unwrap());
        let configs: Vec<McpServerConfig> = vec![];
        let section = build_pinned_resources_section(&registry, &configs, None).await;
        assert!(section.is_empty());
    }

    #[tokio::test]
    async fn pinned_section_skips_unknown_server() {
        let registry = std::sync::Arc::new(McpRegistry::connect_all(&[]).await.unwrap());
        // Server is configured with a pin but never connected (empty registry).
        let configs = vec![McpServerConfig {
            name: "ghost".into(),
            pinned_resources: vec!["file:///x".into()],
            ..Default::default()
        }];
        let section = build_pinned_resources_section(&registry, &configs, None).await;
        // Nothing injected: the read fails/non-existent server is skipped.
        assert!(section.is_empty());
    }

    #[test]
    fn resource_wrapper_labels_untrusted_and_includes_text() {
        let contents = McpResourceContents {
            contents: vec![McpResourceContent {
                uri: "srvA__file:///x".into(),
                mime_type: Some("text/plain".into()),
                text: Some("hello body".into()),
                blob: None,
            }],
        };
        let out = wrap_resource_contents("srvA", "srvA__file:///x", &contents);
        assert!(out.contains("trust=\"untrusted-external\""));
        assert!(out.contains("server=\"srvA\""));
        assert!(out.contains("hello body"));
        assert!(out.starts_with("<mcp-resource"));
        assert!(out.trim_end().ends_with("</mcp-resource>"));
    }

    #[test]
    fn resource_wrapper_redacts_secrets() {
        let contents = McpResourceContents {
            contents: vec![McpResourceContent {
                uri: "srvA__u".into(),
                mime_type: None,
                text: Some("token sk-supersecrettoken12345abcdef end".into()),
                blob: None,
            }],
        };
        let out = wrap_resource_contents("srvA", "srvA__u", &contents);
        assert!(!out.contains("supersecrettoken"), "secret leaked: {out}");
    }

    #[test]
    fn resource_wrapper_notes_blob_without_dumping_bytes() {
        let contents = McpResourceContents {
            contents: vec![McpResourceContent {
                uri: "srvA__b".into(),
                mime_type: Some("application/octet-stream".into()),
                text: None,
                blob: Some("YmFzZTY0".into()),
            }],
        };
        let out = wrap_resource_contents("srvA", "srvA__b", &contents);
        assert!(out.contains("[binary blob"));
        assert!(!out.contains("YmFzZTY0"));
    }

    #[test]
    fn prompt_render_labels_untrusted_and_includes_message_text() {
        let result = McpGetPromptResult {
            description: Some("d".into()),
            messages: vec![McpPromptMessage {
                role: "user".into(),
                content: serde_json::json!({"type":"text","text":"do the thing"}),
            }],
        };
        let out = render_prompt_messages("srvA", "srvA__p", &result);
        assert!(out.contains("trust=\"untrusted-external\""));
        assert!(out.contains("do the thing"));
        assert!(out.contains("user"));
    }
}
