from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from database import get_db
import models
import schemas
from email_config import MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM

router = APIRouter(prefix="/alerts", tags=["Alerts - AlertsPage / AlertsScreen"])


def send_email_alert(recipients: list, subject: str, body: str):
    """Send email to list of recipients using Gmail SMTP."""
    if not recipients:
        return

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)

        for email in recipients:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = MAIL_FROM
            msg["To"]      = email

            # HTML email body
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background: #0f1923; color: #ffffff; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; background: #1a2535; border-radius: 12px; padding: 30px; border: 1px solid #2a3545;">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <h1 style="color: #38bdf8; margin: 0;">IDRMS Alert</h1>
                        <p style="color: #64748b; margin: 5px 0;">Barangay Kauswagan Emergency Alert System</p>
                    </div>
                    <div style="background: #0f1923; border-radius: 8px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #ef4444;">
                        <h2 style="color: #ef4444; margin: 0 0 10px 0;">{subject}</h2>
                        <p style="color: #cbd5e1; line-height: 1.6; margin: 0;">{body}</p>
                    </div>
                    <p style="color: #475569; font-size: 12px; text-align: center; margin: 0;">
                        This is an automated alert from IDRMS — Barangay Kauswagan Disaster Risk Management System.
                    </p>
                </div>
            </body>
            </html>
            """

            msg.attach(MIMEText(html, "html"))
            server.sendmail(MAIL_FROM, email, msg.as_string())

        server.quit()
        print(f"✅ Email sent to {len(recipients)} recipient(s)")

    except Exception as e:
        print(f"❌ Email error: {e}")


@router.get("/", response_model=List[schemas.AlertOut])
def list_alerts(db: Session = Depends(get_db)):
    return db.query(models.Alert).order_by(models.Alert.created_at.desc()).all()


@router.post("/", response_model=schemas.AlertOut, status_code=201)
def create_alert(
    payload: schemas.AlertInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # Save alert to database
    record = models.Alert(**payload.model_dump(exclude={"recipients"}))
    db.add(record)
    db.commit()
    db.refresh(record)

    # Send email in background (non-blocking)
    if payload.recipients:
        subject = f"[IDRMS] {payload.level} Alert – {payload.zone}"
        background_tasks.add_task(
            send_email_alert,
            payload.recipients,
            subject,
            payload.message,
        )

    return record


@router.get("/{alert_id}/", response_model=schemas.AlertOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    record = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return record


@router.delete("/{alert_id}/", status_code=204)
def delete_alert(alert_id: int, db: Session = Depends(get_db)):
    record = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Alert not found.")
    db.delete(record)
    db.commit()
    return None