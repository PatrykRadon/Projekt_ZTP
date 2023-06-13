FROM python:3.8

#WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y cron

# CMD pwd && ls

RUN echo "*/15 * * * * curl http://127.0.0.1:5000/retrain_model" > /etc/cron.d/mycron
RUN chmod 0644 /etc/cron.d/mycron
RUN crontab /etc/cron.d/mycron

COPY start.sh start.sh
RUN chmod +x start.sh

RUN echo "test_1"

EXPOSE 5000

# ENTRYPOINT [ "start.sh" ]
CMD ["sh", "start.sh"]