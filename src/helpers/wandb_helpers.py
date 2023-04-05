from typing import Any, List, Optional

import wandb


def parse_list_to_wandb_table(
    data: List[List[Any]], columns: Optional[List[str]] = None
) -> wandb.Table:
    return wandb.Table(
        columns=columns if columns else [f"col_{i}" for i in range(0, len(data[0]))],
        data=data,
    )
