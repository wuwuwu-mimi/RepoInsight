import random

import typer
from typing_extensions import Annotated


def get_name():
    return random.choice(["Deadpool", "Rick", "Morty", "Hiro"])


def main(name: Annotated[str, typer.Argument(default_factory=get_name, help="The name of the random  user")]):
    print(f"Hello {name}")


def m2(name: Annotated[str, typer.Argument(help="print name",show_default=False)] = "Joker"):
    print(f"Hello {name}")


if __name__ == "__main__":
    typer.run(m2)
