import csv
import click


@click.command()
@click.option(
    "--input_file",
    prompt="Input file name",
    help="Name of the input file",
    required=True,
)
@click.option(
    "--output_file",
    prompt="Output file name",
    help="Name of the output file",
    default="column_converter_output.csv",
)
@click.option(
    "--prompt_column",
    "-p",
    prompt="Prompt column name",
    help="Name of the column to be used as prompt",
    required=True,
)
@click.option(
    "--response_column",
    "-r",
    prompt="Response column name",
    help="Name of the column to be used as response",
    required=True,
)
@click.option(
    "--uid_column",
    "-u",
    prompt="UID column name",
    help="Name of the column to be used as UID",
    required=True,
)
@click.option(
    "--sut_column",
    "-s",
    prompt="SUT column name",
    help="Name of the column to be used as SUT",
    required=True,
)
def main(
    input_file, output_file, prompt_column, response_column, uid_column, sut_column
):
    with open(input_file, "r") as file:
        reader = csv.reader(file)
        columns = next(reader)

        assert prompt_column in columns, f"Column {prompt_column} not found in the file"
        assert (
            response_column in columns
        ), f"Column {response_column} not found in the file"
        assert uid_column in columns, f"Column {uid_column} not found in the file"
        assert sut_column in columns, f"Column {sut_column} not found in the file"

        with open(output_file, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["Prompt", "Response", "UID", "SUT"])

            for row in reader:
                prompt_value = row[columns.index(prompt_column)]
                response_value = row[columns.index(response_column)]
                uid_value = row[columns.index(uid_column)]
                sut_value = row[columns.index(sut_column)]

                if sut_value == "alpaca-7b":
                    writer.writerow(
                        [prompt_value, response_value, uid_value, sut_value]
                    )


if __name__ == "__main__":
    main()
