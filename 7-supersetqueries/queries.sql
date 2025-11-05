-- Clasificación (Gráfico de dona / porcentajes)
SELECT 
    vc.vehicle_name AS classification,
    SUM(v.count) AS total_count,
    ROUND(100.0 * SUM(v.count) / SUM(SUM(v.count)) OVER (), 2) AS percentage
FROM vehicle_count v
JOIN vehicle_class vc ON v.class_id = vc.vehicle_id
GROUP BY vc.vehicle_name
ORDER BY total_count DESC;
-- #########################################################################

-- vehiculos sobre lineas en el tiempo NOt working
SELECT 
    period_date,
    SUM(count) AS total_vehicles
FROM vehicle_count
GROUP BY period_date
ORDER BY period_date;
-- #########################################################################

-- grafico de barras
SELECT 
    vc.vehicle_name AS vehicle_type,
    SUM(v.count) AS total_vehicles
FROM vehicle_count v
JOIN vehicle_class vc ON v.class_id = vc.vehicle_id
GROUP BY vc.vehicle_name
ORDER BY total_vehicles DESC;
