FROM node:22-bookworm-slim AS build
WORKDIR /app
RUN npm install --global pnpm@10.6.5
COPY . .
RUN pnpm install --frozen-lockfile --filter @ocno/oryn-example-editor... \
    && pnpm --filter @ocno/oryn-example-editor build

FROM nginx:stable-alpine
COPY --from=build /app/editor/dist /usr/share/nginx/html
COPY editor/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
