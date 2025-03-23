from rich.table import Table
from rich import get_console


def print_table_metrics(metrics):
    console = get_console()
    table = Table(header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    # console.print(table)


class PrintLastEpochValMetrics:
    """
    A pure Python callback-like object. You can manually call it at the end of training.
    """
    def __init__(self, as_table=True):
        self.as_table = as_table

    def on_validation_end(self, metrics):
        if self.as_table:
            print_table_metrics(metrics)
        else:
            print(metrics)
