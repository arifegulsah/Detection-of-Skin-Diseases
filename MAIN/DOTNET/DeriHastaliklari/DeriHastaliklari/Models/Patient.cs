using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DeriHastaliklari.Models
{
    public class Patient
    {

         
        //Table Data Annotation: Veritabanı tablo adını düzenlemek için kullanılır.
        //Column Data Annotation: Veritabanında yer alan o tablo içindeki kolon adını düzenlemek için kullanılır.

        [Key]
        public int PatientId { get; set; }

        [Required(ErrorMessage ="Lütfen boşlukları doldurunuz")]
        public string Name { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string Surname { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public int TcNo { get; set; }


        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        [DisplayFormat(DataFormatString = "{0:yyyy-MM-dd}", ApplyFormatInEditMode = true)]
        [DataType(DataType.Date)]
        public string bthDay { get; set; }

        //[DataType(DataType.EmailAddress, ErrorMessage = "Lütfen geçerli email adresi girin.")]
        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
       
        public string Email { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string Password { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string Allergy { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string Medicine { get; set; }

        [Required(ErrorMessage = "Lütfen boşlukları doldurunuz")]
        public string Disease { get; set; }

        public AddPhoto AddPhoto { get; set; } //entity olarak ekleniyor.

    }
}
