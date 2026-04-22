import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import dotenv from "dotenv";
import nodemailer from "nodemailer";

dotenv.config();

/**
 * STAMPEDE SHIELD — MISSION CONTROL SERVER
 * Handles automated incident reporting via Gmail/Google SMTP.
 */

// SMTP Configuration
const EMAIL_USER = process.env.EMAIL_USER;
const EMAIL_PASS = process.env.EMAIL_PASS; // App-specific password

const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: EMAIL_USER,
    pass: EMAIL_PASS,
  },
});

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API Route for Incident Reports
  app.post("/api/report", async (req, res) => {
    const { report, recipient, subject, senderEmail, senderPass, isTest } = req.body;

    const timestamp = new Date().toISOString();
    console.log(`\n--- 🚨 ${isTest ? 'TEST CONNECTION' : 'INCOMING INCIDENT'}: ${timestamp} ---`);
    console.log(`RECIPIENT: ${recipient}`);

    const user = (senderEmail || EMAIL_USER)?.toString().trim();
    const pass = (senderPass || EMAIL_PASS)?.toString().replace(/\s+/g, '');

    if (!user || !pass) {
      console.warn(`[AUTH] Credentials missing for ${timestamp}`);
      return res.status(200).json({ 
        status: "logged", 
        message: "No credentials provided. Incident was logged to server logs only." 
      });
    }

    console.log(`[AUTH] Attempting login for ${user} (Pass length: ${pass.length})`);

    try {
      const activeTransporter = nodemailer.createTransport({ 
        service: 'gmail', 
        auth: { user, pass },
        connectionTimeout: 15000, 
        greetingTimeout: 15000 
      });

      // Verify connection early for better error reporting
      if (isTest) {
        await activeTransporter.verify();
      }

      await activeTransporter.sendMail({
        from: `"Stampede Shield" <${user}>`,
        to: recipient,
        subject: subject || (isTest ? "🧪 Stampede Shield: Connection Test" : "🚨 Stampede Shield Incident Alert"),
        text: isTest ? "If you are reading this, your Stampede Shield automated reporting system is correctly configured and capable of sending background alerts." : report,
        html: isTest 
          ? `<div style="font-family: sans-serif; padding: 20px; border: 2px solid #22c55e; border-radius: 8px;">
              <h2 style="color: #22c55e;">✅ Connection Success</h2>
              <p>Your automated incident reporting system is now active.</p>
              <p style="font-size: 0.8rem; color: #64748b;">Timestamp: ${timestamp}</p>
             </div>`
          : `<div style="font-family: sans-serif; max-width: 600px; border: 2px solid #ef4444; border-radius: 12px; overflow: hidden;">
                <div style="background: #ef4444; color: #fff; padding: 20px; font-weight: 800; font-size: 1.2rem;">
                  🚨 CRITICAL INCIDENT DETECTED
                </div>
                <div style="padding: 24px; background: #fff;">
                  <pre style="white-space: pre-wrap; font-family: 'Courier New', Courier, monospace; color: #1e293b;">${report}</pre>
                </div>
              </div>`,
      });

      res.status(200).json({ 
        status: "sent", 
        message: isTest ? "Test email delivered successfully!" : "Automated report dispatched via Google account." 
      });
    } catch (error: any) {
      console.error("Nodemailer Error Details:", error);
      
      let friendlyMessage = error.message;
      if (error.code === 'EAUTH') {
        friendlyMessage = "Google Authentication Failed (535). Please checklist:\n" +
                          "1. 2-Step Verification MUST be ENABLED in your Google account.\n" +
                          "2. You MUST use a 16-character 'App Password' from Google (Security -> 2-Step Verification -> App Passwords).\n" +
                          "3. Do NOT use your regular Gmail login password.";
      } else if (error.code === 'ESOCKET' || error.code === 'ETIMEDOUT') {
        friendlyMessage = "Network Blocked: Direct SMTP traffic (Port 465/587) appears to be restricted in this environment. Please use the 'mailto' fallback in the UI.";
      }

      res.status(500).json({ 
        status: "error", 
        code: error.code,
        message: friendlyMessage
      });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Stampede Shield Server running on http://localhost:${PORT}`);
  });
}

startServer();
