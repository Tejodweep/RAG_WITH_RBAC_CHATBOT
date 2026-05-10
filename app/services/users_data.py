from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


SUPPORTED_ROLES = {
    "administrator",
    "finance",
    "marketing",
    "hr",
    "engineering",
    "c_level",
    "employee",
}


@dataclass(frozen=True)
class DemoUserSeed:
    username: str
    password: str
    role: str


DEFAULT_USERS: Sequence[DemoUserSeed] = (
    DemoUserSeed("Ram", "admin123", "administrator"),
    DemoUserSeed("Tony", "password123", "engineering"),
    DemoUserSeed("Bruce", "securepass", "marketing"),
    DemoUserSeed("Sam", "financepass", "finance"),
    DemoUserSeed("Peter", "pete123", "engineering"),
    DemoUserSeed("Sid", "sidpass123", "marketing"),
    DemoUserSeed("Natasha", "hrpass123", "hr"),
    DemoUserSeed("Morgan", "execpass123", "c_level"),
    DemoUserSeed("Eve", "employeepass", "employee"),
)
