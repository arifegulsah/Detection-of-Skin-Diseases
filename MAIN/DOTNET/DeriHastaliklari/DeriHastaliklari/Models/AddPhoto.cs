using Microsoft.AspNetCore.Http;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DeriHastaliklari.Models
{
    public class AddPhoto
    {
        [Key]
        public int ImageId { get; set; }
        [NotMapped]
        public IFormFile ImageURL { get; set; }

        [Required(ErrorMessage = "Lütfen bir seçim yapınız.")]
        public string Itch { get; set; }

        [Required(ErrorMessage = "Lütfen bir seçim yapınız.")]
        public string Pain { get; set; }

        [Required(ErrorMessage = "Lütfen bir seçim yapınız.")]
        public string Cont { get; set; }
        public string Additional { get; set; }

        [ForeignKey("Patient")]
        public int PatientId { get; set; }

        //PAtient tablosnu buraya tanımlamış oldum.
        public Patient Patient { get; set; }
    }
}
