import pymysql
conn = pymysql.connect(host="127.0.0.1", user="root", password="", database="alltankdata", charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor)
with conn.cursor() as cur:
    for tank in ["tank1", "tank2", "tank3"]:
        cur.execute("SELECT * FROM `" + tank + "` ORDER BY `Timestamp` DESC LIMIT 3")
        rows = cur.fetchall()
        print("=== " + tank + " (latest 3 rows) ===")
        for r in rows:
            print(r)
        print()
conn.close()
