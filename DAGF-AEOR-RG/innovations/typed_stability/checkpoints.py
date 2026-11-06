"""Independent Acc/F1 selection with storage-sharing checkpoint aliases."""
import os
from pathlib import Path
import shutil

from ..target_readout.train import SelectionCheckpoints, save_checkpoint, write_json


def checkpoint_alias(source, destination):
    """Atomically link identical bytes, or copy on filesystems without hard links."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(str(source), str(temporary))
    except OSError:
        shutil.copyfile(str(source), str(temporary))
    temporary.replace(destination)


class BranchCheckpoints(SelectionCheckpoints):
    """Same selection/tie rules as the original; keep raw and EMA independent."""

    def record(self, model, summary, predictions):
        acc_key = (summary["accuracy"], summary["macro_f1"])
        f1_key = (summary["macro_f1"], summary["accuracy"])
        improve_acc = self.best_acc is None or acc_key > (self.best_acc["accuracy"], self.best_acc["macro_f1"])
        improve_f1 = self.best_f1 is None or f1_key > (self.best_f1["macro_f1"], self.best_f1["accuracy"])
        payload = self.payload(model, summary) if improve_acc or improve_f1 else None
        primary = "best_acc" if self.protocol == "baseline" else "best_f1"
        saved = None
        for name, improved in (("best_acc", improve_acc), ("best_f1", improve_f1)):
            if not improved:
                continue
            setattr(self, name, dict(summary))
            destination = self.output / (name + ".pt")
            if saved is None:
                save_checkpoint(destination, payload)
                saved = destination
            else:
                checkpoint_alias(saved, destination)
            write_json(self.output / (name + ".json"), summary)
            write_json(self.output / (name + "_predictions.json"), predictions)
            if name == primary:
                checkpoint_alias(destination, self.output / "best.pt")
        write_json(self.output / "best_scores.json", self.score_summary())
