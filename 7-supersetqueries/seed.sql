-- 1) Insert vehicle classes
INSERT INTO vehicle_class (vehicle_name) VALUES ('Car');
INSERT INTO vehicle_class (vehicle_name) VALUES ('Motorcycle');
INSERT INTO vehicle_class (vehicle_name) VALUES ('Bus');
INSERT INTO vehicle_class (vehicle_name) VALUES ('Truck');
INSERT INTO vehicle_class (vehicle_name) VALUES ('Bicycle');

-- 2) Insert vehicle counts (example periods and counts)
-- We'll assume the UUIDs from vehicle_class are known (replace with actual ones if necessary)
-- To simplify, use subqueries to match by vehicle_name

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Car'), '2025-01-01', 120);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Car'), '2025-02-01', 135);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Car'), '2025-03-01', 142);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Motorcycle'), '2025-01-01', 200);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Motorcycle'), '2025-02-01', 210);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Motorcycle'), '2025-03-01', 190);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Bus'), '2025-01-01', 35);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Bus'), '2025-02-01', 32);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Bus'), '2025-03-01', 37);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Truck'), '2025-01-01', 58);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Truck'), '2025-02-01', 61);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Truck'), '2025-03-01', 64);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Bicycle'), '2025-01-01', 90);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Bicycle'), '2025-02-01', 85);

INSERT INTO vehicle_count (class_id, period_date, count)
VALUES ((SELECT vehicle_id FROM vehicle_class WHERE vehicle_name='Bicycle'), '2025-03-01', 92);
