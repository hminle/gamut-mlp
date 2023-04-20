import csv
from pathlib import Path

from src.utils.reporting.reporter import BaseReporter


class CSVReporter(BaseReporter):
    def __init__(
        self,
        output_dir: str,
        research_name: str,
        dataset_name: str,
        postfix: str,
    ):
        self.output_dir = Path(output_dir)

        self.research_name = research_name
        self.dataset_name = dataset_name
        self.postfix = postfix
        self.table_name = f"{research_name}_{dataset_name}"
        self.data = []

    def set_method_name(self, method_name):
        self.method_name = method_name
        self.output_dir = self.output_dir / self.method_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_table(self) -> None:
        pass

    def create_new_columns(self):
        method_name = self.method_name
        self.fieldnames = [
            "image_filename",
            f"rmse_{method_name}",
            f"psnr_{method_name}",
            f"mae_{method_name}",
            f"deltaE2000_{method_name}",
            f"rmse_oog_{method_name}",
            f"psnr_oog_{method_name}",
            f"training_time_{method_name}",
        ]

    def report_error(
        self,
        image_filename: str,
        rmse: float,
        psnr: float,
        mae: float,
        deltaE2000: float,
        rmse_oog: float,
        psnr_oog: float,
        training_time: float = 0,
    ):
        method_name = self.method_name
        self.data.append(
            {
                "image_filename": image_filename,
                f"rmse_{method_name}": rmse,
                f"psnr_{method_name}": psnr,
                f"mae_{method_name}": mae,
                f"deltaE2000_{method_name}": deltaE2000,
                f"rmse_oog_{method_name}": rmse_oog,
                f"psnr_oog_{method_name}": psnr_oog,
                f"training_time_{method_name}": training_time,
            }
        )

    def report_image(self, image_filename: str, oog_percent: float, idx: int):
        pass

    def stop(self):
        with open(
            Path(self.output_dir) / f"{self.method_name}_{self.table_name}_{self.postfix}.csv",
            "w",
            newline="",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)
