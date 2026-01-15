const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3002;

// Middleware
app.use(cors());
app.use(express.json());

// Statische Dateien (CSS, JS, Images) aus dem aktuellen Ordner servieren
app.use(express.static(path.join(__dirname)));

// Hauptroute
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
    console.log(`
╔═══════════════════════════════════════════════════╗
║         ATLAS LIBRARY SYSTEM ONLINE               ║
╠═══════════════════════════════════════════════════╣
║  ACCESS:   http://localhost:${PORT}                  ║
║  STATUS:   ACTIVE                                 ║
╚═══════════════════════════════════════════════════╝
    `);
});