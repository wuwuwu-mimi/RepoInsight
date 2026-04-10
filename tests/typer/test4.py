import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def create():
    print("create")


@app.command()
def delete(name: str):
    print("delete")


if __name__ == "__main__":
    app()
