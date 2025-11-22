# Security Policy

## Supported Versions
Security fixes are applied to the latest commit on `main`. Older snapshots may not receive updates; pull the newest code before deploying.

## Reporting a Vulnerability
If you discover a security vulnerability, please report it privately:

- Email: `security@datanuri.example`
- Include: Detailed description, steps to reproduce, affected components, and potential impact.

We ask that you do not open a public issue for security reports.

## Response Process
1. We will acknowledge receipt within 3 business days.
2. Maintainers will investigate, reproduce, and determine severity.
3. A fix will be developed, reviewed, and merged as quickly as possible.
4. Once resolved, we will coordinate disclosure and credit the reporter (unless anonymity is requested).

## Best Practices for Deployers
- Keep dependencies up to date (`pip install -r requirements.txt`).
- Restrict network exposure of the FastAPI service unless required.
- Run the backend behind HTTPS and an application firewall when used outside local research environments.
- Rotate API keys or credentials you add to the stack regularly.

