# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""legal-consultant-agent - An Bindu Agent."""

from legal_consultant_agent.__version__ import __version__
from legal_consultant_agent.main import (
    cleanup,
    handler,
    initialize_agent,
    main,
)

__all__ = [
    "__version__",
    "cleanup",
    "handler",
    "initialize_agent",
    "main",
]
