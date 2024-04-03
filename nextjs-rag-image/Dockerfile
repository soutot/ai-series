ARG PNPM_VERSION=8.7.1
FROM node:20.6.1

COPY . ./nextjs-rag-image
WORKDIR /nextjs-rag-image

RUN npm install -g pnpm@${PNPM_VERSION}

ENTRYPOINT pnpm install && pnpm run build && pnpm start