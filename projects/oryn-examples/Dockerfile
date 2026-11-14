FROM node:22-bookworm-slim AS build
WORKDIR /app
RUN npm install --global pnpm@10.6.5
COPY . .
RUN pnpm install --frozen-lockfile
ARG EXAMPLE
RUN pnpm --filter "@ocno/oryn-example-${EXAMPLE}" build \
    && cp -r "${EXAMPLE}/dist" /site

FROM nginx:stable-alpine
COPY --from=build /site /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
