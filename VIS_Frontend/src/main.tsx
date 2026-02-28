import { createRoot } from "react-dom/client";
import { TTSProvider } from "./app/contexts/TTSContext";
import App from "./app/App.tsx";
import "./styles/index.css";

createRoot(document.getElementById("root")!).render(
  <TTSProvider>
    <App />
  </TTSProvider>
);
