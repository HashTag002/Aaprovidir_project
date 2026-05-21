#!/usr/bin/env python
import os
import sys

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aaprovidir_project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Impossible d'importer Django. Est-il installé et disponible dans ton PYTHONPATH ? "
            "As-tu activé ton environnement virtuel ?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()