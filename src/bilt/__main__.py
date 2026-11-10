"""Allow python -m bilt check as well as the console script."""

from bilt._cli import main

raise SystemExit(main())
