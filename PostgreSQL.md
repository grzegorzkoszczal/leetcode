# Cookbook for using PostgreSQL on Linux

## Installation

### Ubuntu

Tutorial:\
https://www.postgresql.org/docs/current/install-make.html

### Arch Linux

```
sudo pacman -Syu
sudo pacman -S postgresql
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql
sudo -u postgres createuser --interactive
sudo -u postgres createdb <database_name>
sudo -u postgres psql
```

## Basic concepts

### Storing data

Data is stored in the tables.
Each table is a named collections of rows.
ach row of a given table has the same set of named columns, and each column is of a specific data type.
Tables are grouped into databases.
Collection of databases managed by a single PostgreSQL server instance constitutes a database cluster.

### Commands

Change user to postgres in order to work on database

```
sudo su postgres
```

Creating database:

```
createdb <database_name>
```

Deleting database:

```
dropdb <database_name>
```

Start working on the database:

```
psql <database_name>
```

Creating a new table:

```
CREATE TABLE weather (
    city            varchar(80),
    temp_lo         int,           -- low temperature
    temp_hi         int,           -- high temperature
    prcp            real,          -- precipitation
    date            date
);
```

Removing table:

```
DROP TABLE tablename;
```

Populating a table with rows:
(This approach requires You to remember the order of the columns)

```
INSERT INTO weather VALUES ('San Francisco', 46, 50, 0.25, '1994-11-27');
```

An alternative syntax allows you to list the columns explicitly:

```
INSERT INTO weather (city, temp_lo, temp_hi, prcp, date)
    VALUES ('San Francisco', 43, 57, 0.0, '1994-11-29');
```

In order to copy large amount of data from flat-text files, it is advised to use COPY command, because of the optimization its faster than INSERT

```
COPY weather FROM '/home/user/weather.txt';
```

#### Querying a table

To retrieve data from a table, the table is queried. An SQL SELECT statement is used to do this.

Retrieving all the rows of table weather:

```
SELECT * FROM weather;
```