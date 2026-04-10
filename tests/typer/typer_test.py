import typer
from rich import print


data = {
    "name": "Rick",
    "age": 42,
    "items": [{"name": "Portal Gun"}, {"name": "Plumbus"}],
    "active": True,
    "affiliation": None,
}


def main():
    print("[bold red]Alert![/bold red] [green]Portal gun[/green] shooting! :boom:")


if __name__ == "__main__":
    typer.run(main)