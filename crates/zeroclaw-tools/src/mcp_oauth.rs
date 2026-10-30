//! OAuth 2.1 for remote MCP servers (`oauth = true` on an `http`/`sse` server).
//!
//! The MCP authorization flow: the server's protected-resource metadata names its authorization
//! server; ZeroClaw registers itself there (dynamic client registration), signs in with the
//! authorization-code flow and PKCE, and keeps the token. `zeroclaw mcp login <server>` runs the
//! sign-in once; it works on a headless host: it prints the sign-in URL, and the user pastes back
//! the address the browser lands on afterwards (a `http://127.0.0.1:…` page that won't load).
//! Transports then send `Authorization: Bearer <token>`, refreshed when the server issued a
//! refresh token. Tokens live in `<config dir>/mcp-oauth/<server>.json`, readable by the owner
//! only.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The address the authorization server sends the browser back to. Nothing listens there: the
/// user copies it from the browser (headless hosts have no browser to receive it).
const REDIRECT_URI: &str = "http://127.0.0.1:33418/callback";
/// Refresh this long before the token expires.
const EXPIRY_MARGIN_SECS: i64 = 60;

/// What `login` stores for a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredToken {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Unix seconds; absent when the server gave no lifetime.
    #[serde(default)]
    pub expires_at: Option<i64>,
    pub token_endpoint: String,
    pub client_id: String,
    #[serde(default)]
    pub client_secret: Option<String>,
    /// The MCP server URL the token is for (RFC 8707 resource indicator).
    pub resource: String,
}

/// `<config dir>/mcp-oauth`, the config directory resolved like the config itself
/// (`ZEROCLAW_CONFIG_DIR`, else `~/.zeroclaw`).
fn token_dir() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("ZEROCLAW_CONFIG_DIR") {
        let custom = custom.trim();
        if !custom.is_empty() {
            return Ok(PathBuf::from(custom).join("mcp-oauth"));
        }
    }
    let home = std::env::var("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".zeroclaw").join("mcp-oauth"))
}

fn token_path(server: &str) -> Result<PathBuf> {
    let safe: String = server
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    Ok(token_dir()?.join(format!("{safe}.json")))
}

fn load(server: &str) -> Result<Option<StoredToken>> {
    let path = token_path(server)?;
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).with_context(|| format!("reading {}", path.display()))?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

fn save(server: &str, token: &StoredToken) -> Result<()> {
    let path = token_path(server)?;
    let dir = path.parent().context("token path has no parent")?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let bytes = serde_json::to_vec_pretty(token)?;
    let tmp = path.with_extension("json.tmp");
    #[cfg(unix)]
    {
        use std::io::Write as _;
        use std::os::unix::fs::OpenOptionsExt as _;
        let mut file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&tmp)?;
        file.write_all(&bytes)?;
    }
    #[cfg(not(unix))]
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
}

/// The bearer token for `server`: the stored one, refreshed first when it is about to expire.
/// `Ok(None)` when the server was never signed in to.
pub async fn bearer(server: &str, client: &reqwest::Client) -> Result<Option<String>> {
    let Some(mut token) = load(server)? else {
        return Ok(None);
    };
    let expiring = token.expires_at.is_some_and(|at| at - EXPIRY_MARGIN_SECS <= now());
    if expiring {
        let Some(refresh) = token.refresh_token.clone() else {
            bail!("the OAuth token for MCP server `{server}` has expired: run `zeroclaw mcp login {server}`");
        };
        let mut form = vec![
            ("grant_type", "refresh_token".to_string()),
            ("refresh_token", refresh),
            ("client_id", token.client_id.clone()),
            ("resource", token.resource.clone()),
        ];
        if let Some(secret) = &token.client_secret {
            form.push(("client_secret", secret.clone()));
        }
        let response = client.post(&token.token_endpoint).form(&form).send().await.context("refreshing the MCP OAuth token")?;
        if !response.status().is_success() {
            bail!(
                "refreshing the OAuth token for MCP server `{server}` failed (HTTP {}): run `zeroclaw mcp login {server}`",
                response.status()
            );
        }
        let fresh: TokenResponse = response.json().await.context("reading the refreshed token")?;
        token.access_token = fresh.access_token;
        if fresh.refresh_token.is_some() {
            token.refresh_token = fresh.refresh_token;
        }
        token.expires_at = fresh.expires_in.map(|secs| now() + secs);
        save(server, &token)?;
    }
    Ok(Some(token.access_token))
}

/// Adds `Authorization: Bearer <token>` for an OAuth server (`server` is `Some`); a server never signed
/// in to is an error that says how to sign in.
pub async fn authorize(
    request: reqwest::RequestBuilder,
    server: Option<&str>,
    client: &reqwest::Client,
) -> Result<reqwest::RequestBuilder> {
    let Some(server) = server else {
        return Ok(request);
    };
    match bearer(server, client).await? {
        Some(token) => Ok(request.bearer_auth(token)),
        None => bail!("MCP server `{server}` uses OAuth and is not signed in: run `zeroclaw mcp login {server}`"),
    }
}

fn b64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// A random string of URL-safe characters (two v4 UUIDs: 244 random bits).
fn random_string() -> String {
    format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple())
}

fn pkce_challenge(verifier: &str) -> String {
    b64url(&Sha256::digest(verifier.as_bytes()))
}

/// `https://host:port` of a URL.
fn origin_of(url: &reqwest::Url) -> String {
    let mut origin = format!("{}://{}", url.scheme(), url.host_str().unwrap_or_default());
    if let Some(port) = url.port() {
        origin.push_str(&format!(":{port}"));
    }
    origin
}

async fn get_json(client: &reqwest::Client, url: &str) -> Result<Option<serde_json::Value>> {
    let response = client.get(url).header("Accept", "application/json").send().await?;
    if !response.status().is_success() {
        return Ok(None);
    }
    Ok(response.json().await.ok())
}

/// The `resource_metadata` URL from a `WWW-Authenticate: Bearer resource_metadata="…"` header.
fn resource_metadata_from_challenge(header: &str) -> Option<String> {
    let start = header.find("resource_metadata=")? + "resource_metadata=".len();
    let rest = header[start..].trim_start_matches('"');
    let end = rest.find(['"', ',', ' ']).unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

/// The authorization server's metadata for the MCP server at `resource`.
async fn discover(client: &reqwest::Client, resource: &str) -> Result<serde_json::Value> {
    let url = reqwest::Url::parse(resource).context("the MCP server URL is not a URL")?;
    let origin = origin_of(&url);

    // 1. Protected-resource metadata: from the 401 challenge, else the well-known locations.
    let mut candidates = Vec::new();
    let probe = client
        .post(resource)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .body(r#"{"jsonrpc":"2.0","id":0,"method":"ping"}"#)
        .send()
        .await
        .context("reaching the MCP server")?;
    if let Some(challenge) = probe.headers().get("WWW-Authenticate").and_then(|v| v.to_str().ok()) {
        if let Some(metadata) = resource_metadata_from_challenge(challenge) {
            candidates.push(metadata);
        }
    }
    let path = url.path().trim_end_matches('/');
    candidates.push(format!("{origin}/.well-known/oauth-protected-resource{path}"));
    candidates.push(format!("{origin}/.well-known/oauth-protected-resource"));
    let mut issuer = None;
    for candidate in &candidates {
        if let Some(metadata) = get_json(client, candidate).await? {
            issuer = metadata
                .get("authorization_servers")
                .and_then(|servers| servers.get(0))
                .and_then(|server| server.as_str())
                .map(str::to_string);
            if issuer.is_some() {
                break;
            }
        }
    }
    let issuer = issuer.unwrap_or(origin);

    // 2. Authorization-server metadata (RFC 8414, then OpenID Connect discovery).
    let issuer_url = reqwest::Url::parse(&issuer).context("the authorization server is not a URL")?;
    let issuer_origin = origin_of(&issuer_url);
    let issuer_path = issuer_url.path().trim_end_matches('/');
    for candidate in [
        format!("{issuer_origin}/.well-known/oauth-authorization-server{issuer_path}"),
        format!("{issuer_origin}/.well-known/openid-configuration{issuer_path}"),
        format!("{}/.well-known/openid-configuration", issuer.trim_end_matches('/')),
    ] {
        if let Some(metadata) = get_json(client, &candidate).await? {
            if metadata.get("authorization_endpoint").is_some() && metadata.get("token_endpoint").is_some() {
                return Ok(metadata);
            }
        }
    }
    bail!("no OAuth authorization server metadata found for {resource} (issuer {issuer})")
}

/// Signs in to the MCP server `server` at `resource`, reading the redirect address from `input`
/// after printing the sign-in URL to `output`. Stores the token for `bearer`.
pub async fn login(
    server: &str,
    resource: &str,
    client: &reqwest::Client,
    input: &mut (dyn std::io::BufRead + Send),
    output: &mut (dyn std::io::Write + Send),
) -> Result<()> {
    let metadata = discover(client, resource).await?;
    let field = |name: &str| metadata.get(name).and_then(|v| v.as_str()).map(str::to_string);
    let authorization_endpoint = field("authorization_endpoint").context("no authorization_endpoint")?;
    let token_endpoint = field("token_endpoint").context("no token_endpoint")?;
    let registration_endpoint = field("registration_endpoint")
        .context("the authorization server offers no dynamic client registration")?;

    let registration = client
        .post(&registration_endpoint)
        .json(&serde_json::json!({
            "client_name": "ZeroClaw",
            "redirect_uris": [REDIRECT_URI],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }))
        .send()
        .await
        .context("registering with the authorization server")?;
    if !registration.status().is_success() {
        let status = registration.status();
        let body = registration.text().await.unwrap_or_default();
        bail!("client registration failed (HTTP {status}): {}", body.chars().take(300).collect::<String>());
    }
    let registration: serde_json::Value = registration.json().await?;
    let client_id = registration.get("client_id").and_then(|v| v.as_str()).context("no client_id in the registration")?.to_string();
    let client_secret = registration.get("client_secret").and_then(|v| v.as_str()).map(str::to_string);

    let verifier = random_string();
    let state = random_string();
    let scope = metadata
        .get("scopes_supported")
        .and_then(|v| v.as_array())
        .map(|scopes| scopes.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    let mut authorize = reqwest::Url::parse(&authorization_endpoint).context("bad authorization_endpoint")?;
    {
        let mut query = authorize.query_pairs_mut();
        query
            .append_pair("response_type", "code")
            .append_pair("client_id", &client_id)
            .append_pair("redirect_uri", REDIRECT_URI)
            .append_pair("code_challenge", &pkce_challenge(&verifier))
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", &state)
            .append_pair("resource", resource);
        if !scope.is_empty() {
            query.append_pair("scope", &scope);
        }
    }

    writeln!(output, "Open this address in a browser and approve the sign-in:\n\n{authorize}\n")?;
    writeln!(output, "The browser then lands on {REDIRECT_URI}?code=… (the page won't load). Paste that whole address here:")?;
    output.flush()?;
    let mut line = String::new();
    input.read_line(&mut line)?;
    let landed = reqwest::Url::parse(line.trim()).context("that is not the address the browser landed on")?;
    let param = |name: &str| landed.query_pairs().find(|(k, _)| k == name).map(|(_, v)| v.into_owned());
    if let Some(error) = param("error") {
        bail!("the sign-in was refused: {error}");
    }
    if param("state").as_deref() != Some(state.as_str()) {
        bail!("the address doesn't belong to this sign-in (state differs)");
    }
    let code = param("code").context("no code in the address")?;

    let mut form = vec![
        ("grant_type", "authorization_code".to_string()),
        ("code", code),
        ("redirect_uri", REDIRECT_URI.to_string()),
        ("client_id", client_id.clone()),
        ("code_verifier", verifier),
        ("resource", resource.to_string()),
    ];
    if let Some(secret) = &client_secret {
        form.push(("client_secret", secret.clone()));
    }
    let response = client.post(&token_endpoint).form(&form).send().await.context("exchanging the code")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("the token exchange failed (HTTP {status}): {}", body.chars().take(300).collect::<String>());
    }
    let token: TokenResponse = response.json().await.context("reading the token")?;
    save(
        server,
        &StoredToken {
            access_token: token.access_token,
            refresh_token: token.refresh_token,
            expires_at: token.expires_in.map(|secs| now() + secs),
            token_endpoint,
            client_id,
            client_secret,
            resource: resource.to_string(),
        },
    )?;
    writeln!(output, "Signed in to MCP server `{server}`; the token is in {}.", token_path(server)?.display())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_matches_rfc_7636_example() {
        assert_eq!(pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    }

    #[test]
    fn reads_resource_metadata_from_a_challenge() {
        let header = r#"Bearer error="invalid_token", resource_metadata="https://t3.example/.well-known/oauth-protected-resource""#;
        assert_eq!(
            resource_metadata_from_challenge(header).as_deref(),
            Some("https://t3.example/.well-known/oauth-protected-resource")
        );
        assert!(resource_metadata_from_challenge("Bearer realm=\"x\"").is_none());
    }

    #[test]
    fn random_strings_are_pkce_safe_and_distinct() {
        let a = random_string();
        assert!((43..=128).contains(&a.len()));
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric()));
        assert_ne!(a, random_string());
    }

    #[test]
    fn server_names_become_file_names() {
        let path = token_path("t3/../x").unwrap();
        assert_eq!(path.file_name().unwrap(), "t3____x.json");
    }
}
