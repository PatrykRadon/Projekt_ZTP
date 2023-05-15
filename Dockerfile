FROM python:3.8

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y cron

RUN echo "*/15 * * * * curl http://127.0.0.1:5000/retrain_model" > /etc/cron.d/mycron
RUN chmod 0644 /etc/cron.d/mycron
RUN crontab /etc/cron.d/mycron

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 5000

CMD ["/app/start.sh"]