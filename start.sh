#!/bin/sh
service cron start
flask run --host 0.0.0.0 --port 5000