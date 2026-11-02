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

#[test]
fn deepseek_dsml_calls_parse_into_their_tool_calls() {
    // As OpenCode passed them through on 2026-11-02, doubled U+FF5C bars.
    let observed = "<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name=\"concierge__series_list\">\n\n</｜｜DSML｜｜ invoke>\n<｜｜DSML｜｜ invoke name=\"remote__devices_list\">\n\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>";
    let (visible, calls) = parse_tool_calls(observed);
    assert!(visible.trim().is_empty(), "{visible:?}");
    let names: Vec<&str> = calls.iter().map(|call| call.name.as_str()).collect();
    assert_eq!(names, ["concierge__series_list", "remote__devices_list"]);
    assert_eq!(calls[0].arguments, serde_json::json!({}));

    // DeepSeek's documented shape: string and JSON parameters.
    let documented = "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"concierge__series_get\">\n<｜DSML｜parameter name=\"tvdbId\" string=\"false\">331753</｜DSML｜parameter>\n<｜DSML｜parameter name=\"seasons\" string=\"false\">[2]</｜DSML｜parameter>\n<｜DSML｜parameter name=\"quality\" string=\"true\">1080p</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>";
    let (_, calls) = parse_tool_calls(documented);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].arguments,
        serde_json::json!({ "tvdbId": 331753, "seasons": [2], "quality": "1080p" })
    );

    // ASCII bars, with words before the call.
    let ascii = "Checking.\n<|DSML|invoke name=\"remote__devices_list\"></|DSML|invoke>";
    let (visible, calls) = parse_tool_calls(ascii);
    assert_eq!(visible, "Checking.");
    assert_eq!(calls[0].name, "remote__devices_list");
}
