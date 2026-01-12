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
    handler,
    initialize_agent,
    main,
    cleanup,
)

__all__ = [
    "__version__",
    "handler",
    "initialize_agent",
    "main",
    "cleanup",
]