# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 20:17:34 2022

@author: arife
"""

# pip install pypyodbc

import pypyodbc


#Windows Authentication İle Bağlantı
db = pypyodbc.connect(
    'Driver={SQL Server};'
    'Server=DESKTOP-TVM6RQR\SQLEXPRESS;'
    'Database=DeriHastalik;'
    'Trusted_Connection=True;'
)

#SQL Server Authentication İle Bağlantı
db = pypyodbc.connect(
    'Driver={SQL Server};'
    'Server=DESKTOP-TVM6RQR\SQLEXPRESS;'
    'Database=DeriHastalik;'
    'UID=kubra;'
    'PWD=12345678;'
)

imlec = db.cursor()


imlec.execute('SELECT * FROM Kisiler')

#diziyi değişkende tutuyoruz.
#!! Tek bir satır çekmek için fetchall() yerine fetchone() fonksiyonunu kullanırız.
kullanicilar = imlec.fetchall()

#yazdırmak istersek
for i in kullanicilar:
    print(i)


"""

#INSERT ISLEMI
komut = 'INSERT INTO Kisiler VALUES(?,?,?)'
veriler = ('Özlem','ÖZ',25)

sonuc = imlec.execute(komut,veriler)
db.commit()

"""

#UPDATE ISLEMI
sonuc = imlec.execute('UPDATE Kisiler SET yas = ? WHERE id = ?',(50,1))
db.commit()

print(str(sonuc) + " kullanıcı güncellendi")
