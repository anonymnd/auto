# GitHub Actions setup

The workflow in `.github/workflows/rabat-stage-brief.yml` checks the local
Casablanca time hourly and sends the Rabat internship brief at 08:00. This
handles Morocco timezone changes automatically.

## Repository secrets

In the GitHub repository, open **Settings > Secrets and variables > Actions**.
Create these repository secrets:

| Secret | Value |
| --- | --- |
| `GMAIL_SENDER` | The Gmail address used to send the brief |
| `GMAIL_APP_PASSWORD` | The Gmail App Password, without spaces |
| `BRIEF_RECIPIENT` | The email address that receives the brief |

Never commit the Gmail App Password to the repository.

## Test the workflow

Open **Actions > Rabat internship morning brief > Run workflow**. The workflow
will send an email even when no matching LinkedIn post is found.

## Filters

The script keeps only public LinkedIn posts that:

- were published during the last 7 days;
- mention Rabat and an IT-related topic;
- mention July as the intended internship period;
- are not PFE or end-of-studies internships.
