import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { ttsRouter } from './routes/tts.js';
import { sttRouter } from './routes/stt.js';

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({ origin: true }));
app.use(express.json({ limit: '50mb' })); // Increased limit for audio data

app.use('/api', ttsRouter);
app.use('/api', sttRouter);

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'vis-backend' });
});

app.listen(PORT, () => {
  console.log(`VIS Backend running at http://localhost:${PORT}`);
});
