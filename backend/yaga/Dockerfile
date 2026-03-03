FROM ruby:2.7.8-slim

ENV APP_HOME=/app \
    BUNDLE_PATH=/usr/local/bundle \
    BUNDLE_JOBS=4 \
    BUNDLE_RETRY=3 \
    BUNDLE_FORCE_RUBY_PLATFORM=true

WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libxml2-dev \
    libxslt1-dev \
    liblzma-dev \
    zlib1g-dev \
    pkg-config \
    netcat-openbsd \
    nodejs \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN gem install bundler -v 1.17.3

COPY Gemfile Gemfile.lock ./
RUN bundle _1.17.3_ install

COPY . .
RUN chmod +x docker-entrypoint.sh

EXPOSE 3000

CMD ["./docker-entrypoint.sh"]
