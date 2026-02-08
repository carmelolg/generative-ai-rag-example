# Security Policy

## Supported Versions

This project currently supports the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Dependency Management

### Python Dependencies

This project maintains up-to-date dependencies to minimize security vulnerabilities:

- **Python Version**: 3.10 or higher (recommended: 3.12+)
- **Dependency Updates**: Automated via Dependabot (weekly checks)
- **Security Scanning**: Automated via GitHub Actions using `pip-audit`

### Current Dependencies

- `ollama~=0.6.1` - Ollama Python client for LLM interactions
- `python-dotenv~=1.2.1` - Environment variable management

### Version Constraints

Dependencies use compatible release specifiers (`~=`) to:
- Allow automatic patch version updates (bug fixes, security patches)
- Prevent breaking changes from minor version updates
- Balance security with stability

## Vulnerability Reporting

If you discover a security vulnerability in this project:

1. **DO NOT** open a public issue
2. Email the maintainer directly (see repository owner information)
3. Provide detailed information about the vulnerability
4. Allow reasonable time for a fix before public disclosure

## Security Best Practices

When using this project:

1. **Environment Variables**: Never commit `.env` files or expose API keys
2. **Dependencies**: Regularly update dependencies using `pip install -U -r requirements.txt`
3. **Python Version**: Use Python 3.10+ for security patches and modern features
4. **Virtual Environments**: Always use a virtual environment to isolate dependencies
5. **Ollama**: Keep your Ollama installation up to date

## Automated Security Measures

This repository implements:

1. **Dependabot**: Automatically opens PRs for dependency updates
2. **GitHub Actions**: Weekly security scans using `pip-audit`
3. **Dependency Pinning**: Compatible release constraints to prevent breaking changes
4. **Code Scanning**: Can be enabled via GitHub Advanced Security

## Manual Security Checks

To manually check for vulnerabilities:

```bash
# Install security scanning tools
pip install pip-audit

# Scan project dependencies
pip-audit -r requirements.txt

# Or scan installed packages
pip-audit
```

## Security Update Process

1. Dependabot or manual scanning identifies a vulnerability
2. Automated PR is created with the fix
3. CI/CD runs security scans on the PR
4. Maintainer reviews and merges if tests pass
5. New version is released if applicable

## Additional Resources

- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [GitHub Dependabot](https://docs.github.com/en/code-security/dependabot)
