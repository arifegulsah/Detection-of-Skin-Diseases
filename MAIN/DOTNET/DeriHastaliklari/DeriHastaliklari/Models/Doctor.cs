using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DeriHastaliklari.Models
{
    public class Doctor
    {
        [Key]
        public int DoctorID { get; set; }

        [Column(TypeName ="Varchar(20)")]
        public string Username { get; set; }

        [Column(TypeName = "Varchar(20)")]
        public string Password { get; set; }
    }
}
