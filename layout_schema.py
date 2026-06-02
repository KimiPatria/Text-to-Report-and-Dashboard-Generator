from typing import Literal, Optional

from pydantic import BaseModel, Field


class WidgetConfig(BaseModel):
    id: str
    chart_type: Literal["bar", "line", "pie", "big_number", "table", "scatter"]
    title: str
    x_field: Optional[str] = None
    y_field: Optional[str] = None
    group_by: Optional[str] = None
    table: str
    sql_hint: Optional[str] = None   # SELECT skeleton; validated but not executed
    grid: dict = Field(default_factory=lambda: {"x": 0, "y": 0, "w": 6, "h": 4})


class DashboardLayout(BaseModel):
    title: str
    widgets: list[WidgetConfig]
    tables_used: list[str]


class LayoutRequest(BaseModel):
    goal: str
    max_widgets: int = 6
