using Microsoft.EntityFrameworkCore;

namespace DeriHastaliklari.Models
{
    public class Context : DbContext
    {
        // veritabanı oluşturuyoruz authorize işlemleri için startupa gidip cookieyi ekliyoruz.
        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlServer("server=DESKTOP-UNSRF7H\\SQLEXPRESS; database=DeriHastalik; integrated security=true");
        }

        public DbSet<Doctor> Doctors { get; set; }
        public DbSet<Patient> Patients { get; set; }
    }
}
