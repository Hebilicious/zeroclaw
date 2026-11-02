//! Tool calls a model wrote as text in a native-tool turn (seen from Claude Haiku through the
//! gateway). Those that name a tool must parse into that call, so the turn runs it; one that
//! names none must be recognised as malformed protocol, so the turn retries. Neither may be
//! delivered as the reply.

use std::collections::HashSet;
use zeroclaw_tool_call_parser::{
    looks_like_malformed_tool_protocol_envelope_for_known_tools, parse_tool_calls,
};

fn known() -> HashSet<String> {
    ["concierge__series_get", "concierge__movie_list"]
        .into_iter()
        .map(String::from)
        .collect()
}

#[test]
fn observed_named_parameter_calls_parse_into_their_tool_calls() {
    let observed = [
        (
            "<parameter\n<parameter name=\"name\">concierge__series_get</parameter>\n<parameter name=\"arguments\">{\"tvdbId\": 331753, \"seasons\": [2]}</parameter>\n</invoke>",
            "concierge__series_get",
            serde_json::json!({ "tvdbId": 331753, "seasons": [2] }),
        ),
        (
            "<invoke_tool>\n<parameter name=\"name\">concierge__movie_list</parameter>\n<parameter name=\"arguments\">{}</parameter>\n</invoke_tool>",
            "concierge__movie_list",
            serde_json::json!({}),
        ),
        (
            "<invoke/>\n<parameter name=\"tool\">concierge__movie_list</parameter>\n</invoke>",
            "concierge__movie_list",
            serde_json::json!({}),
        ),
    ];
    for (text, name, arguments) in observed {
        let (visible, calls) = parse_tool_calls(text);
        assert!(visible.trim().is_empty(), "{text:?} left text {visible:?}");
        assert_eq!(calls.len(), 1, "{text:?}");
        assert_eq!(calls[0].name, name);
        assert_eq!(calls[0].arguments, arguments);
    }
}

#[test]
fn observed_placeholder_reply_is_malformed_protocol() {
    let text = "_call_placeholder\n\n_call_placeholder\n\n_call_placeholder\n\n_call_placeholder";
    assert!(looks_like_malformed_tool_protocol_envelope_for_known_tools(
        text,
        &known()
    ));
    assert!(
        !looks_like_malformed_tool_protocol_envelope_for_known_tools(
            "a call_placeholder in prose",
            &known()
        )
    );
}

#[test]
fn observed_empty_tool_tags_are_malformed_protocol() {
    let text = "<invoke_web_search_placeholder>\n</invoke_web_search_placeholder>";
    assert!(looks_like_malformed_tool_protocol_envelope_for_known_tools(
        text,
        &known()
    ));
    assert!(
        !looks_like_malformed_tool_protocol_envelope_for_known_tools(
            "<b>Season 2</b> is out",
            &known()
        )
    );
}

#[test]
fn prose_quoting_the_markup_is_not_a_call() {
    let text = "Write <parameter name=\"name\">x</parameter> to name the tool.";
    let (visible, calls) = parse_tool_calls(text);
    assert!(calls.is_empty(), "{calls:?}");
    assert!(!visible.is_empty());
}
