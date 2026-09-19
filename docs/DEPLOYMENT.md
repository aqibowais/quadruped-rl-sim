# Deployment and public access

This application needs a Python process continuously running MuJoCo and
PyTorch. It cannot be deployed as a GitHub Pages, Netlify, or Vercel static
site. Those services can host the HTML, CSS, and JavaScript, but they cannot
run the simulator or the two reinforcement-learning trainers.

## Option 1: free public demo from your computer

This is the most practical zero-cost option for a portfolio review or
interview. Cloudflare Quick Tunnels provide a temporary public HTTPS URL while
the app continues to run on your computer.

### One-time setup

Install Cloudflare's tunnel client:

```powershell
winget install Cloudflare.cloudflared
```

Close and reopen the terminal after installation.

### Start the public demo

From the repository root:

```powershell
.\scripts\public_demo.ps1
```

The terminal prints a URL similar to:

```text
https://random-words.trycloudflare.com
```

Anyone with that URL can open the live application. Keep the terminal and
computer running. Press `Ctrl+C` to stop both the tunnel and the Python server.
The URL changes each time because Quick Tunnels are temporary.

The script sets `ALLOW_RESTART=0`, so public visitors cannot discard the
networks with the **Restart training** button. The application is still a
shared live session: all visitors see the same trainers and display
simulations.

### Appropriate use

Use a Quick Tunnel for:

- a live interview demonstration;
- a short portfolio review;
- sharing with a professor or teammate for a limited period.

Do not treat it as permanent hosting. Your computer supplies all compute, and
the URL disappears when the process stops.

## Option 2: persistent container deployment

The repository includes a production-oriented `Dockerfile`. It installs the
CPU-only PyTorch wheel, starts the server on `0.0.0.0:7860`, runs as a
non-root user, disables public training restarts, and provides a health check.

Build and test it locally:

```powershell
docker build -t quadruped-rl-live .
docker run --rm -p 7860:7860 quadruped-rl-live
```

Open `http://127.0.0.1:7860`.

Any container host can run this image if it provides approximately:

- 2 CPU cores minimum;
- 4 GB RAM minimum, with 8 GB preferred;
- a continuously running web process;
- support for server-sent events;
- a configurable public port.

CPU hosting is functional but slow. A GPU host improves SAC training
throughput, but the supplied image intentionally uses CPU-only PyTorch for
portability.

### Hugging Face Spaces

Docker Spaces use port `7860`, which already matches this repository. Create a
Docker Space, then push this repository to the Space's Git remote. The Space
README metadata should contain:

```yaml
---
title: PPO vs SAC Live Training
sdk: docker
app_port: 7860
---
```

As of September 2026, Hugging Face requires a paid account plan to create
Docker or CPU-compute Spaces, even though CPU Basic itself has no hourly
hardware charge. Static Spaces remain free, but a static Space cannot run this
backend. Check the current Spaces pricing before selecting this route.

### Other hosts

Render, Railway, Fly.io, and similar providers can run the Dockerfile, but
their current free or trial tiers change frequently. A 512 MB instance is not
large enough for MuJoCo, Stable-Baselines3, two policies, and a SAC replay
buffer. Verify memory, process-sleep, and billing limits before deploying.

Oracle Cloud's Always Free VM can be suitable when capacity is available, but
account creation may require payment-card verification and the VM must be
configured and secured manually.

## Make the GitHub repository public

The configured remote is:

```text
https://github.com/aqibowais/quadruped-rl-sim.git
```

Push the prepared commit:

```powershell
git push -u origin master
```

If Git asks for authentication, use GitHub Desktop, Git Credential Manager, or
a personal access token. GitHub does not accept account passwords for Git
pushes.

Then open the repository on GitHub:

1. Select **Settings**.
2. Open **General**.
3. Scroll to **Danger Zone**.
4. Select **Change repository visibility**.
5. Choose **Public** and confirm the repository name.

Never commit API tokens, passwords, `.env` files, private keys, replay
buffers, or local credentials. The `.gitignore` excludes generated models and
large replay buffers.

## Why model artifacts are not in Git

The local SAC replay buffers are roughly 454 MB each, and the full local model
directory is over 9 GB. GitHub blocks ordinary Git files over 100 MB and
recommends keeping repositories small. These are generated experiment
artifacts, not source code.

For a public release of trained weights, upload only final model archives to a
GitHub Release, Hugging Face model repository, or object store. Do not publish
every replay-buffer checkpoint. Include a checksum, training configuration,
seed, and evaluation result with each released model.

## Production limitations

This project is a research demonstration, not a multi-tenant service:

- PPO and SAC are process-local and are lost when the process restarts.
- All visitors share the same training state.
- Pause, reset, speed, and policy selections affect the shared display.
- There is no authentication or persistent database.
- Free hosts may sleep, reset storage, or throttle CPU.

A true public service would move each training job to a queue, isolate jobs in
workers, persist model snapshots to object storage, add authentication and
rate limiting, and give each browser a session-specific display environment.
