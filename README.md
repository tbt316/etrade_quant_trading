# E*TRADE API Python Sample Application

This sample Python application provides examples on using the ETRADE API endpoints.

> **Safety status:** live order execution, service installation, and remote
> restart are intentionally disabled while the durable execution migration is
> incomplete. The repository is not yet approved for unattended trading.

## Repository and Credential Hygiene

- Create virtual environments locally; `venv/`, `.venv/`, package metadata,
  caches, logs, generated reports, and runtime state are not source files and
  must remain untracked.
- Keep OAuth values, broker configuration, dashboard settings, session files,
  and arming material only in ignored owner-readable local files. Commit only
  placeholder examples such as `.env.example`.
- Run `python etrade_python_client/scripts/check_repo_hygiene.py` before a
  commit. CI applies the same deterministic check to Git's index.
- Ignore rules and removal from the current index do not erase old Git history.
  Previously exposed broker keys still require external revocation/rotation
  and coordinated history cleanup.

## Table of Contents

* [Requirements](#requirements)
* [Setup](#setup)
* [Running Code](#running-code)

## Requirements

In order to run this sample application you need the following three items:

1. Python 3 - this sample application is written in Python and requires Python 3. If you do not
already have Python 3 installed, download it from

   [`https://www.python.org/downloads/`](https://www.python.org/downloads/).

2. An [E*TRADE](https://us.etrade.com) account

3. E*TRADE consumer key and consumer secret.


## Setup

1. Unzip python zip file

2. Create a local, ignored `etrade_python_client/config.ini` and add the
consumer key and consumer secret from the E*TRADE application keys page. Never
commit that file.

3. Create the virtual environment by running the Python's venv command; see the command syntax below

```
$ python3 -m venv venv
```

4. Activate the Python virtual environment

On Windows, run:

```
$ venv\Scripts\activate.bat
```

On Unix or Mac OS, run:

```
$ source venv/bin/activate
```

5. Use pip to install dependencies for the sample application

```
$ pip install -r requirements.txt
```

6. Run the sample application

```
$ cd etrade_python_client
$ python3 etrade_python_client.py
```

## Running Code

Complete these steps to run the code for the sample application:

1. Activate the Python virtual environment

On Windows, run:

```
$ venv\Scripts\activate.bat
```

On Unix or Mac OS, run:

```
$ source venv/bin/activate
```

2. Run the application

```
$ cd etrade_python_client
$ python3 etrade_python_client.py
```
