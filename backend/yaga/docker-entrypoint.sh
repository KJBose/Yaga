#!/bin/sh
set -e

: "${MONGODB_URI:=mongodb://mongodb:27017/yaga}"

MONGO_HOSTPORT=$(echo "${MONGODB_URI}" | sed -E 's|^mongodb://([^/@]+@)?([^/?]+).*|\2|')
MONGO_HOST=$(echo "${MONGO_HOSTPORT}" | cut -d: -f1)
MONGO_PORT=$(echo "${MONGO_HOSTPORT}" | cut -d: -f2)

if [ -z "${MONGO_PORT}" ] || [ "${MONGO_PORT}" = "${MONGO_HOSTPORT}" ]; then
  MONGO_PORT=27017
fi

echo "Waiting for MongoDB at ${MONGO_HOST}:${MONGO_PORT}..."
until nc -z "${MONGO_HOST}" "${MONGO_PORT}"; do
  sleep 1
done

echo "Starting Rails API server..."
exec bundle _1.17.3_ exec rails server -b 0.0.0.0 -p 3000
