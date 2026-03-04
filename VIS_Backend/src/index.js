import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { ttsRouter } from './routes/tts.js';

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({ origin: true }));
app.use(express.json());

app.use('/api', ttsRouter);

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'vis-backend' });
});

app.listen(PORT, () => {
  console.log(`VIS Backend running at http://localhost:${PORT}`);
});