const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');

// Since Playwright isn't easily accessible, let's try using puppeteer or a different approach
// For now, let's fetch the HTML and analyze it
fetch('http://127.0.0.1:5077/')
  .then(res => res.text())
  .then(html => {
    console.log('HTML length:', html.length);
    console.log('Has .brand-mark:', html.includes('brand-mark'));
    console.log('Has .score-cell:', html.includes('score-cell'));
  })
  .catch(err => console.error(err));
