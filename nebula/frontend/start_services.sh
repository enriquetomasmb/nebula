#!/bin/bash

# Print commands and their arguments as they are executed (debugging)
set -x

# Print in console debug messages
echo "Starting services..."

# Start Nginx in the foreground in a subshell for the script to proceed
nginx &

# Change directory to where app.py is located
NEBULA_FRONTEND_DIR=/nebula/nebula/frontend
cd $NEBULA_FRONTEND_DIR

# Start Gunicorn
NEBULA_SOCK=nebula.sock
echo "DEV: $NEBULA_DEV"
if [ "$NEBULA_DEV" = "True" ]; then
    echo "Starting Gunicorn in development mode..."
    NEBULA_SOCK=nebula.sock
fi

DEBUG=$NEBULA_DEBUG
echo "DEBUG: $DEBUG"
if [ "$DEBUG" = "True" ]; then
    echo "Starting Gunicorn in debug mode..."
    gunicorn --worker-class eventlet --workers 1 --bind unix:/tmp/$NEBULA_SOCK --access-logfile $SERVER_LOG --error-logfile $SERVER_LOG  --reload --reload-extra-file $NEBULA_FRONTEND_DIR --capture-output --log-level debug app:app &
else
    echo "Starting Gunicorn in production mode..."
    gunicorn --worker-class eventlet --workers 1 --bind unix:/tmp/$NEBULA_SOCK --access-logfile $SERVER_LOG app:app &
fi

tensorboard --host 0.0.0.0 --port 8080 --logdir $NEBULA_LOGS_DIR --window_title "NEBULA Statistics" --reload_interval 30 --max_reload_threads 10 --reload_multifile true &

# Start statistics dashboard
# aim init --repo $NEBULA_LOGS_DIR
# --dev flag is used to enable development mode
# aim server --repo $NEBULA_LOGS_DIR --port 8085 &
# aim up --repo $NEBULA_LOGS_DIR --port 8080 --base-path /statistics --dev &

tail -f /dev/null
