import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { PrismaClient } from '@prisma/client';

dotenv.config();

const app = express();
const prisma = new PrismaClient();
const PORT = process.env.PORT || 4000;

app.use(cors());
app.use(express.json());

// Main App Routes
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'app-backend' });
});

// Auth Routes (Scaffold)
app.post('/api/auth/register', async (req, res) => {
  res.status(501).json({ error: 'Not implemented' });
});

app.post('/api/auth/login', async (req, res) => {
  res.status(501).json({ error: 'Not implemented' });
});

// Avatar Routes
app.get('/api/users/:userId/avatar', async (req, res) => {
  res.status(501).json({ error: 'Not implemented' });
});

app.listen(PORT, () => {
  console.log(`App backend listening on port ${PORT}`);
});
