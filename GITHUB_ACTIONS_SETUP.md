# GitHub Actions setup

The workflow in [`.github/workflows/rabat-stage-brief.yml`](</C:/Users/PC/Documents/New project/.github/workflows/rabat-stage-brief.yml>) is the GitHub-only delivery path for the Rabat internship brief.
It runs on GitHub-hosted Ubuntu and sends the email from GitHub Actions, without relying on the local PowerShell script or a Windows credential file.

## Repository secrets

In the GitHub repository, open **Settings > Secrets and variables > Actions** and create these repository secrets:

| Secret | Value |
| --- | --- |
| `GMAIL_SENDER` | The Gmail address used to send the brief |
| `GMAIL_APP_PASSWORD` | The Gmail App Password, without spaces |
| `BRIEF_RECIPIENT` | The email address that receives the brief |

Never commit the Gmail App Password to the repository. The workflow now fails fast if one of these secrets is missing.

## Test the workflow

Open **Actions > Rabat internship morning brief > Run workflow**.

Expected behavior:

- The email is sent by GitHub Actions, not by the local machine.
- The received message includes `Source d'envoi: GitHub Actions`.
- The received message includes the GitHub run URL for traceability.
- The workflow summary in GitHub contains the generated brief text.

## Important behavior

- The scheduled workflow only marks the day as sent after a real email send.
- A scheduled run skipped because of the Casablanca hour no longer blocks the next valid run.
- The local script [`send-stage-brief-email.ps1`](</C:/Users/PC/Documents/New project/send-stage-brief-email.ps1>) is no longer required for the GitHub delivery path.

## Filters

The script keeps only public LinkedIn posts that:

- were published during the last 7 days;
- mention Rabat and an IT-related topic;
- mention July as the intended internship period;
- are not PFE or end-of-studies internships.
