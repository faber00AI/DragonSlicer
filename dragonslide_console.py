"""Console entry of the portable build:  dragonSlide-cli.exe cli <tool> ...   |   dragonSlide-cli.exe mcp   (stdio MCP server)."""
import sys


def main():
    mode = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1] in ("cli", "mcp") else None
    if mode == "mcp":
        import dragonslide_mcp
        dragonslide_mcp.server.run("stdio")
    elif mode == "cli":
        from dragonslide_cli import main as cli
        sys.exit(cli())
    else:
        print("uso: dragonSlide-cli.exe cli <strumento> [opzioni]   (elenco: dragonSlide-cli.exe cli --help)\n"
              "     dragonSlide-cli.exe mcp                            (server MCP su stdio)")
        sys.exit(2)


if __name__ == "__main__":
    main()
