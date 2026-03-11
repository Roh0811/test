# GitHub Copilot + Figma Integration

This repository is configured to connect GitHub Copilot with Figma, enabling AI-assisted development based on Figma designs.

## Setup Instructions

### 1. Get Your Figma Access Token

1. Go to [Figma Account Settings](https://www.figma.com/settings)
2. Scroll down to **Personal access tokens**
3. Click **Generate new token**
4. Copy the generated token

### 2. Configure the Access Token

#### For Repository Secrets (Recommended for Teams)
1. Go to your repository's **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `FIGMA_ACCESS_TOKEN`
4. Value: Paste your Figma access token
5. Click **Add secret**

#### For Local Development
Create a `.env` file in your project root (make sure it's in `.gitignore`):
```
FIGMA_ACCESS_TOKEN=your_figma_access_token_here
```

Or set it as an environment variable:
```bash
export FIGMA_ACCESS_TOKEN=your_figma_access_token_here
```

## Usage

Once configured, you can use GitHub Copilot to:

- **Extract design information**: Ask Copilot about colors, typography, spacing, and components from your Figma files
- **Generate code from designs**: Describe a Figma component and get code suggestions
- **Sync design tokens**: Extract design tokens from Figma and generate CSS/SCSS variables
- **Create component stubs**: Generate component code based on Figma layer structure

### Example Prompts

```
"Get the colors used in my Figma file [URL]"
"Generate a React component based on the Button component in Figma"
"Extract the typography styles from my Figma design system"
"What are the spacing values used in this Figma frame?"
```

## Configuration

The MCP server configuration is located in `.github/copilot/mcp.json`. It uses the `@anthropic/figma-mcp-server` package to provide Figma integration capabilities.

## Troubleshooting

### Token Issues
- Ensure your Figma access token has the correct permissions
- Check that the token hasn't expired
- Verify the token is correctly set in your secrets or environment

### Connection Issues
- Make sure you have Node.js installed (for npx)
- Check your network connection to Figma's API
- Verify the Figma file URL is correct and accessible with your token

## Security Notes

- Never commit your Figma access token directly in code
- Use environment variables or GitHub secrets for token management
- Regularly rotate your access tokens for security

## Learn More

- [Figma API Documentation](https://www.figma.com/developers/api)
- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [MCP Server Protocol](https://modelcontextprotocol.io/)
