# Security Policy

## API Keys and Secrets

### NEVER commit to git:
- API keys
- Passwords
- Tokens
- Private keys
- `.env` files

### How to handle secrets:

1. **Local Development**
   ```bash
   cp .env.example .env
   # Edit .env with your actual keys
   ```

2. **Production / CI/CD**
   - Use GitHub Secrets
   - Use environment variables
   - Use secret management (AWS Secrets Manager, etc.)

3. **Code Review Checklist**
   - [ ] No hardcoded credentials
   - [ ] `.env` in `.gitignore`
   - [ ] Use `config/settings.py` for all config access
   - [ ] Log statements don't leak secrets

## Current Secrets

| Service | Env Var | Status |
|---------|---------|--------|
| Salad Cloud | `SALAD_CLOUD_API_KEY` | Required for GPU deployments |
| GitHub | `GITHUB_TOKEN` | Required for git push |

## Incident Response

If a secret is accidentally committed:
1. Rotate the key immediately
2. Remove from git history (force push or use BFG)
3. Notify service provider
4. Update this document
