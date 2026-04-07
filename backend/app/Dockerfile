FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
COPY tsconfig.json ./
RUN npm install

COPY . .

EXPOSE 4000

CMD ["npx", "ts-node", "server.ts"]
