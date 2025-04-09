#!/bin/bash
set -e

export USER_NAME=docker_user

# Get the container's UID/GID
USER_ID=${USER_ID:-9001}
GROUP_ID=${GROUP_ID:-9001}
USER_NAME=${USER_NAME:-appuser}

# Add the group and user if they don't exist
if ! getent group $GROUP_ID >/dev/null; then
  echo "Creating group $USER_NAME with GID $GROUP_ID"
  groupadd -g $GROUP_ID $USER_NAME
fi

if ! getent passwd $USER_ID >/dev/null; then
  echo "Creating user $USER_NAME with UID $USER_ID"
  useradd -u $USER_ID -g $GROUP_ID -s /bin/bash -m -d /home/$USER_NAME $USER_NAME
fi

# If running as root, switch to the new user
if [ "$(id -u)" = "0" ]; then
  echo "Switching to user $USER_NAME"
  exec gosu $USER_NAME "$@"
else
  exec "$@"
fi
